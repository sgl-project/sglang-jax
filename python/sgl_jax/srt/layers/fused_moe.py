"""Fused Expert-Parallel MoE layer using Pallas kernel."""

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig, fused_ep_moe
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor


def _expand_moe_block_scale(scale_3d: jax.Array, n_out: int, block_n: int) -> jax.Array:
    """Expand compact 2D MoE block scales to the kernel's fast 1D-ready layout."""
    scale_per_channel = jnp.repeat(scale_3d, block_n, axis=2)[..., :n_out]
    return scale_per_channel[:, :, None, :]


class FusedEPMoE(nnx.Module):
    """
    Expert Parallel MoE layer using fused TPU kernel.

    This layer wraps the optimized fused_ep_moe kernel which combines Top-K selection,
    expert computation, and aggregation into a single efficient operation.

    Key differences from EPMoE:
    - Weight format: w1/w3 are (num_experts, hidden_size, intermediate_size) for gate/up proj
      and w2 is (num_experts, intermediate_size, hidden_size) for down proj
    - Input: Takes router_logits directly instead of pre-computed topk_weights/topk_ids
    - Implementation: Uses Pallas kernel with manual memory management for TPU optimization

    Args:
        hidden_size: Hidden size of the model
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts to select per token (top_k)
        ep_size: Expert parallel size (number of devices to shard experts across)
        mesh: JAX mesh for distributed execution
        intermediate_dim: Intermediate dimension for expert FFN
        weight_dtype: Data type for weights
        dtype: Data type for computation
        activation: Activation function ("silu", "gelu", "swigluoai")
        layer_id: Layer index (for debugging)
        renormalize_topk_logits: Whether to renormalize top-k weights
        bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c: Tile size parameters (auto-selected if None)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        use_grouped_topk: bool = False,
        num_groups: int = 1,
        top_k_groups: int = 1,
        renormalize_topk_logits: bool = False,
        routed_scaling_factor: float | None = None,
        num_shared_experts: int = 0,
        moe_shared_expert_intermediate_size: int | None = None,
        quantization_config=None,
        enable_act_quant: bool = True,
        # Profiling / ablation flags (primarily for microbenching).
        disable_a2a: bool = False,
        disable_dynamic_ffn1: bool = False,
        disable_dynamic_ffn2: bool = False,
        disable_weight_load: bool = False,
        disable_a2a_s_tile_read: bool = False,
        disable_a2a_s_acc_tile_write: bool = False,
        disable_shared_expert: bool = False,
        disable_all_reduce_metadata: bool = False,
        disable_sync_barrier: bool = False,
        use_jax_allreduce_metadata: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.activation = activation
        self.use_grouped_topk = use_grouped_topk
        self.num_groups = num_groups
        self.top_k_groups = top_k_groups
        self.renormalize_topk_logits = renormalize_topk_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.num_shared_experts = num_shared_experts
        self.moe_shared_expert_intermediate_size = (
            moe_shared_expert_intermediate_size or intermediate_dim
        )
        self.mesh = mesh
        self.disable_a2a = disable_a2a
        self.disable_dynamic_ffn1 = disable_dynamic_ffn1
        self.disable_dynamic_ffn2 = disable_dynamic_ffn2
        self.disable_weight_load = disable_weight_load
        self.disable_a2a_s_tile_read = disable_a2a_s_tile_read
        self.disable_a2a_s_acc_tile_write = disable_a2a_s_acc_tile_write
        self.disable_shared_expert = disable_shared_expert
        self.disable_all_reduce_metadata = disable_all_reduce_metadata
        self.disable_sync_barrier = disable_sync_barrier
        self.use_jax_allreduce_metadata = use_jax_allreduce_metadata

        metadata = get_global_expert_location_metadata()
        if metadata is not None and layer_id is not None:
            self.num_experts = metadata.num_physical_experts
        else:
            self.num_experts = num_experts

        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )

        self.quantized_dtype = (
            quantization_config.get_moe_weight_dtype() if quantization_config else None
        )
        self.activation_quantized_dtype = (
            quantization_config.get_moe_activation_dtype() if quantization_config else None
        )
        # Optional explicit disable for in-kernel activation quantization. The
        # positive signal comes from quantization_config.moe_activation_dtype.
        self.enable_act_quant_cfg = enable_act_quant

        # Initialize weights.
        self.w1 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (self.num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )
        self.w3 = nnx.Param(
            jax.random.normal(
                jax.random.key(1),
                (self.num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w2 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (self.num_experts, intermediate_dim, hidden_size),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w1_scale = None
        self.w3_scale = None
        self.w2_scale = None

        if self.num_shared_experts > 0:
            se_inter_dim = self.moe_shared_expert_intermediate_size * self.num_shared_experts

            self.w1_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w2_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (se_inter_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w3_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )
        else:
            self.w1_shared = None
            self.w3_shared = None
            self.w2_shared = None

        self.w1_shared_scale = None
        self.w3_shared_scale = None
        self.w2_shared_scale = None

        # Read block-wise quantization settings from config.
        weight_block_size = (
            getattr(quantization_config, "weight_block_size", None) if quantization_config else None
        )
        if weight_block_size is not None and len(weight_block_size) == 2:
            self.quant_block_k = int(weight_block_size[1])  # block_k
            self.quant_block_n = int(weight_block_size[0])  # block_n
        else:
            self.quant_block_k = None
            self.quant_block_n = None

    def quantize_weights(self, is_static: bool = False):
        """Quantize MoE weights in-place. Call once after model loading."""
        if self.quantized_dtype is None:
            return

        # Determine quant_block_k. The v1 kernel requires a block size when
        # scales are provided, so per-channel fp8 must tile to block-256; the v2
        # kernel accepts per-channel (None).
        wsz = (
            self.quant_block_k
            if self.quant_block_k is not None
            else (None if isinstance(self, FusedEPMoEV2) else 256)
        )
        if hasattr(self, "quant_block_k"):
            del self.quant_block_k
        self.quant_block_k = wsz

        with jax.set_mesh(self.mesh):
            if is_static:
                ep_scale_sharding = P(("data", "tensor"), None, None, None)

                if wsz is None:
                    # Per-channel: scale shape (E, 1, 1, N)
                    w1_scale_shape = (
                        self.num_experts,
                        1,
                        1,
                        self.intermediate_dim,
                    )
                    w3_scale_shape = w1_scale_shape
                    w2_scale_shape = (
                        self.num_experts,
                        1,
                        1,
                        self.hidden_size,
                    )
                else:
                    # Block-wise: scale shape (E, K//block_k, 1, N)
                    w1_scale_shape = (
                        self.num_experts,
                        self.hidden_size // wsz,
                        1,
                        self.intermediate_dim,
                    )
                    w3_scale_shape = w1_scale_shape
                    w2_scale_shape = (
                        self.num_experts,
                        self.intermediate_dim // wsz,
                        1,
                        self.hidden_size,
                    )

                if hasattr(self, "w1_scale"):
                    del self.w1_scale
                self.w1_scale = nnx.Param(
                    jnp.zeros(w1_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if hasattr(self, "w3_scale"):
                    del self.w3_scale
                self.w3_scale = nnx.Param(
                    jnp.zeros(w3_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if hasattr(self, "w2_scale"):
                    del self.w2_scale
                self.w2_scale = nnx.Param(
                    jnp.zeros(w2_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if self.num_shared_experts > 0:
                    # fused kernel expects per-channel shared scale (1, 1, se_inter)
                    # — see _validate_fused_ep_moe_args / kernel.py:455-470
                    shared_scale_sharding = P(None, None, None)
                    se_inter = self.moe_shared_expert_intermediate_size * self.num_shared_experts

                    if hasattr(self, "w1_shared_scale"):
                        del self.w1_shared_scale
                    self.w1_shared_scale = nnx.Param(
                        jnp.zeros((1, 1, se_inter), dtype=jnp.float32),
                        out_sharding=shared_scale_sharding,
                    )

                    if hasattr(self, "w3_shared_scale"):
                        del self.w3_shared_scale
                    self.w3_shared_scale = nnx.Param(
                        jnp.zeros((1, 1, se_inter), dtype=jnp.float32),
                        out_sharding=shared_scale_sharding,
                    )

                    if hasattr(self, "w2_shared_scale"):
                        del self.w2_shared_scale
                    self.w2_shared_scale = nnx.Param(
                        jnp.zeros((1, 1, self.hidden_size), dtype=jnp.float32),
                        out_sharding=shared_scale_sharding,
                    )

                return

            # Replace original weights with quantized versions
            if self.quant_block_n is not None:
                # 2D block-wise quantization: scale shape (E, K//block_k, N//block_n)
                w1_value, w1_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
                w3_value, w3_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
                w2_value, w2_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
            else:
                # 1D sub-channel quantization: scale shape (E, K//wsz, N)
                w1_value, w1_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )
                w3_value, w3_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )
                w2_value, w2_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )

            # NOTE: Fused MoE shards the expert dimension across EP=(data*tensor).
            ep_sharding = P(("data", "tensor"), None, None)
            ep_scale_sharding = P(("data", "tensor"), None, None, None)

            self.w1 = nnx.Param(w1_value, out_sharding=ep_sharding)
            self.w3 = nnx.Param(w3_value, out_sharding=ep_sharding)
            self.w2 = nnx.Param(w2_value, out_sharding=ep_sharding)

            # Update scales (reshape to 4D for GMM kernel)
            if self.quant_block_n is not None:
                # 2D block-wise: expand block scales once so forward can run
                # through the fast 1D kernel path without changing semantics.
                w1_scale_4d = _expand_moe_block_scale(
                    w1_scale, self.intermediate_dim, self.quant_block_n
                )
                w3_scale_4d = _expand_moe_block_scale(
                    w3_scale, self.intermediate_dim, self.quant_block_n
                )
                w2_scale_4d = _expand_moe_block_scale(
                    w2_scale, self.hidden_size, self.quant_block_n
                )
            else:
                if self.quant_block_k is None:
                    # Per-channel reduction over K returns [E, N].
                    w1_scale_4d = w1_scale[:, None, None, :]
                    w3_scale_4d = w3_scale[:, None, None, :]
                    w2_scale_4d = w2_scale[:, None, None, :]
                else:
                    # Sub-channel [E, K//wsz, N] -> [E, K//wsz, 1, N].
                    w1_scale_4d = w1_scale[:, :, None, :]
                    w3_scale_4d = w3_scale[:, :, None, :]
                    w2_scale_4d = w2_scale[:, :, None, :]

            if hasattr(self, "w1_scale"):
                del self.w1_scale
            self.w1_scale = nnx.Param(
                w1_scale_4d,
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w3_scale"):
                del self.w3_scale
            self.w3_scale = nnx.Param(
                w3_scale_4d,
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w2_scale"):
                del self.w2_scale
            self.w2_scale = nnx.Param(
                w2_scale_4d,
                out_sharding=ep_scale_sharding,
            )

            if self.w1_shared is not None:
                w1_shared_value, w1_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1_shared.value,
                    axis=0,
                )
                w3_shared_value, w3_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3_shared.value,
                    axis=0,
                )
                w2_shared_value, w2_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2_shared.value,
                    axis=0,
                )

                self.w1_shared = nnx.Param(w1_shared_value, out_sharding=P(None, None))
                self.w3_shared = nnx.Param(w3_shared_value, out_sharding=P(None, None))
                self.w2_shared = nnx.Param(w2_shared_value, out_sharding=P(None, None))

                if hasattr(self, "w1_shared_scale"):
                    del self.w1_shared_scale
                self.w1_shared_scale = nnx.Param(
                    w1_shared_scale.reshape(
                        1,
                        1,
                        w1_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w3_shared_scale"):
                    del self.w3_shared_scale
                self.w3_shared_scale = nnx.Param(
                    w3_shared_scale.reshape(
                        1,
                        1,
                        w3_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w2_shared_scale"):
                    del self.w2_shared_scale
                self.w2_shared_scale = nnx.Param(
                    w2_shared_scale.reshape(
                        1,
                        1,
                        w2_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

    def __call__(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        block_config: FusedMoEBlockConfig | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Forward pass through the fused MoE layer."""
        assert hidden_states.ndim == 2

        use_shared_expert = self.w1_shared is not None and not self.disable_shared_expert
        w1_shared_val = self.w1_shared.value if use_shared_expert else None
        w3_shared_val = self.w3_shared.value if use_shared_expert else None
        w2_shared_val = self.w2_shared.value if use_shared_expert else None
        w1_scale = self.w1_scale.value if self.w1_scale is not None else None
        w3_scale = self.w3_scale.value if self.w3_scale is not None else None
        w2_scale = self.w2_scale.value if self.w2_scale is not None else None
        w1_shared_scale = (
            self.w1_shared_scale.value
            if use_shared_expert and self.w1_shared_scale is not None
            else None
        )
        w3_shared_scale = (
            self.w3_shared_scale.value
            if use_shared_expert and self.w3_shared_scale is not None
            else None
        )
        w2_shared_scale = (
            self.w2_shared_scale.value
            if use_shared_expert and self.w2_shared_scale is not None
            else None
        )

        quant_block_k = self.quant_block_k if self.quant_block_k is not None else None

        kernel_sharding = jax.sharding.NamedSharding(self.mesh, P(("data", "tensor"), None))
        hidden_states = jax.sharding.reshard(hidden_states, kernel_sharding)
        topk_weights = jax.sharding.reshard(topk_weights, kernel_sharding)
        topk_ids = jax.sharding.reshard(topk_ids, kernel_sharding)

        output = fused_ep_moe(
            mesh=self.mesh,
            tokens=hidden_states,
            w1=self.w1.value,
            w2=self.w2.value,
            w3=self.w3.value,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=self.num_experts_per_tok,
            use_grouped_topk=self.use_grouped_topk,
            num_groups=self.num_groups,
            top_k_groups=self.top_k_groups,
            renormalize_topk_logits=self.renormalize_topk_logits,
            routed_scaling_factor=self.routed_scaling_factor,
            act_fn=self.activation,
            block_config=block_config,
            disable_a2a=self.disable_a2a,
            disable_dynamic_ffn1=self.disable_dynamic_ffn1,
            disable_dynamic_ffn2=self.disable_dynamic_ffn2,
            disable_weight_load=self.disable_weight_load,
            disable_a2a_s_tile_read=self.disable_a2a_s_tile_read,
            disable_a2a_s_acc_tile_write=self.disable_a2a_s_acc_tile_write,
            disable_shared_expert=self.disable_shared_expert,
            disable_all_reduce_metadata=self.disable_all_reduce_metadata,
            disable_sync_barrier=self.disable_sync_barrier,
            use_jax_allreduce_metadata=self.use_jax_allreduce_metadata,
            # Optional parameters (not used in basic case)
            quant_block_k=quant_block_k,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=w1_shared_val,
            w2_shared=w2_shared_val,
            w3_shared=w3_shared_val,
            w1_shared_scale=w1_shared_scale,
            w2_shared_scale=w2_shared_scale,
            w3_shared_scale=w3_shared_scale,
            b1=None,
            b2=None,
            b3=None,
            dp_axis_name="data",
            tp_axis_name="tensor",
        )

        if out_sharding is None:
            out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * output.ndim)))
        output = jax.sharding.reshard(output, out_sharding)
        return output


class FusedEPMoEV2(FusedEPMoE):
    """V2 fused EP-MoE layer using the Strix-style double-buffer kernel.

    Inherits weight init and quantization from FusedEPMoE. Overrides __call__
    to dispatch to fused_ep_moe_v2 with v2-specific flags.
    """

    def __call__(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        block_config=None,
        out_sharding: jax.sharding.Sharding | None = None,
        swiglu_limit: float | None = None,
        shared_swiglu_limit: float | None = None,
    ) -> jax.Array:
        from sgl_jax.srt.kernels.fused_moe.v2.kernel import fused_ep_moe_v2
        from sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs import (
            get_tuned_fused_moe_v2_block_config,
        )

        assert hidden_states.ndim == 2

        w1_scale = self.w1_scale.value if self.w1_scale is not None else None
        w3_scale = self.w3_scale.value if self.w3_scale is not None else None
        w2_scale = self.w2_scale.value if self.w2_scale is not None else None

        use_shared_expert = self.w1_shared is not None and not self.disable_shared_expert
        w1_shared_val = self.w1_shared.value if use_shared_expert else None
        w3_shared_val = self.w3_shared.value if use_shared_expert else None
        w2_shared_val = self.w2_shared.value if use_shared_expert else None

        # SE per-channel scales are stored 3D (1, 1, out); the v2 kernel reads them
        # 2D (1, out). Squeeze here (not in quantize_weights, which the v1 path and
        # the weight mapping consume in 3D form). None for bf16 SE weights (Mode 3).
        w1_shared_scale = (
            self.w1_shared_scale.value[:, 0, :]
            if use_shared_expert and self.w1_shared_scale is not None
            else None
        )
        w3_shared_scale = (
            self.w3_shared_scale.value[:, 0, :]
            if use_shared_expert and self.w3_shared_scale is not None
            else None
        )
        w2_shared_scale = (
            self.w2_shared_scale.value[:, 0, :]
            if use_shared_expert and self.w2_shared_scale is not None
            else None
        )
        # In-kernel act-quant (fp8 token, Mode 1) needs both a quant-config
        # activation dtype and fp8 weights. Weight-only fp8 checkpoints stay in
        # Mode 2 (bf16 token x fp8 weight).
        enable_act_quant = (
            self.activation_quantized_dtype is not None and self.enable_act_quant_cfg is not False
        ) and (w1_scale is not None)
        quant_block_k = self.quant_block_k if hasattr(self, "quant_block_k") else None
        quant_mode = (
            "none"
            if w1_scale is None
            else ("per_channel" if quant_block_k is None else "blockwise")
        )

        if block_config is None:
            block_config = get_tuned_fused_moe_v2_block_config(
                num_tokens=hidden_states.shape[0],
                num_experts=self.num_experts,
                top_k=self.num_experts_per_tok,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_dim,
                dtype=hidden_states.dtype,
                weight_dtype=self.w1.value.dtype,
                ep_size=self.ep_size,
                use_shared_expert=use_shared_expert,
                use_grouped_topk=self.use_grouped_topk,
                enable_act_quant=enable_act_quant,
                quant_mode=quant_mode,
            )

        direct_scaled_dot = w1_scale is not None
        kernel_sharding = jax.sharding.NamedSharding(self.mesh, P(("data", "tensor"), None))
        hidden_states = jax.sharding.reshard(hidden_states, kernel_sharding)
        topk_weights = jax.sharding.reshard(topk_weights, kernel_sharding)
        topk_ids = jax.sharding.reshard(topk_ids, kernel_sharding)

        output = fused_ep_moe_v2(
            self.mesh,
            hidden_states,
            self.w1.value,
            self.w2.value,
            self.w3.value,
            topk_weights,
            topk_ids,
            self.num_experts_per_tok,
            act_fn=self.activation,
            swiglu_limit=swiglu_limit,
            shared_swiglu_limit=shared_swiglu_limit,
            block_config=block_config,
            quant_block_k=quant_block_k,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=w1_shared_val,
            w2_shared=w2_shared_val,
            w3_shared=w3_shared_val,
            w1_shared_scale=w1_shared_scale,
            w2_shared_scale=w2_shared_scale,
            w3_shared_scale=w3_shared_scale,
            enable_act_quant=enable_act_quant,
            direct_scaled_dot=direct_scaled_dot,
            dp_axis_name="data",
            tp_axis_name="tensor",
        )

        # Reshard the MoE output to the caller-requested layout. Under sequence
        # parallelism out_sharding carries the SP-aware reduce_sharding
        # (('data','tensor') on the scatter dim); without it we fall back to the
        # plain DP layout P('data', None). b79f9951 dropped this arg and hardcoded
        # the DP layout, which silently broke SP for MoE layers — restored here.
        if out_sharding is not None:
            output = jax.sharding.reshard(output, out_sharding)
        else:
            output = jax.sharding.reshard(
                output, jax.sharding.NamedSharding(self.mesh, P("data", None))
            )
        return output


def fused_rs_shared_expert(
    hidden_states: jax.Array,
    w1: jax.Array,
    w3: jax.Array,
    w2: jax.Array,
    *,
    w1_scale: jax.Array | None,
    w3_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    activation_quantized_dtype: jnp.dtype | None,
    mesh: Mesh,
    v2_shared_block_size: int = 1024,
) -> jax.Array:
    """Run GLM's replicated shared expert beside the routed RS path."""
    from sgl_jax.srt.kernels.fused_moe.fused_rs.fused_moe_rs import get_moe_expert_axis

    expert_axis = get_moe_expert_axis(mesh)
    token_spec = P(expert_axis, None)
    hidden_states = jax.sharding.reshard(hidden_states, NamedSharding(mesh, token_spec))

    def _local_linear(
        x,
        weight,
        scale,
    ):
        if scale is None:
            return jax.lax.dot_general(
                x.astype(jnp.float32),
                weight.astype(jnp.float32),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

        scale_f32 = jnp.reshape(scale, (1, -1)).astype(jnp.float32)
        if activation_quantized_dtype is None:
            return jax.lax.dot_general(
                x.astype(jnp.float32),
                weight.astype(jnp.float32) * scale_f32,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

        # V2 explicitly promotes the bf16 source to f32 before amax/scale.
        # quantize_tensor_simple computes those reductions in the source dtype,
        # so using it here rounds the token scale to bf16 before both FFNs.
        x_f32 = x.astype(jnp.float32)
        dtype_max = jnp.float32(jnp.finfo(activation_quantized_dtype).max)
        x_amax = jnp.max(jnp.abs(x_f32), axis=-1, keepdims=True)
        x_scale = jnp.maximum(x_amax / dtype_max, jnp.float32(1e-12))
        x_q = (x_f32 / x_scale).astype(activation_quantized_dtype)
        acc = jax.lax.dot_general(
            x_q,
            weight,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        # Keep the stable f32 scale operation order without reproducing V2's
        # lossy in-band scale encoding, which overwrites four FP8 activation
        # channels before FFN1.
        combined_scale = x_scale.astype(jnp.float32) * scale_f32
        return acc.astype(jnp.float32) * combined_scale

    def _run(x, w1_local, w3_local, w2_local, s1, s3, s2):
        gate = _local_linear(x, w1_local, s1)
        up = _local_linear(x, w3_local, s3)
        intermediate = jax.nn.silu(gate) * up
        if v2_shared_block_size <= 0:
            raise ValueError(f"v2_shared_block_size must be positive, got {v2_shared_block_size}")

        # V2's EP32/64K config uses bse=1024.  It quantizes every shared FFN2
        # input block independently and rounds each accumulated partial back to
        # the bf16 output buffer.  A single full-intermediate quantization has a
        # different scale and produces ~3% layer drift on the real checkpoint.
        block_size = min(v2_shared_block_size, intermediate.shape[1])
        output = None
        for start in range(0, intermediate.shape[1], block_size):
            end = min(start + block_size, intermediate.shape[1])
            partial = _local_linear(
                intermediate[:, start:end],
                w2_local[start:end, :],
                s2,
            )
            if output is None:
                output = partial.astype(x.dtype)
            else:
                output = (output.astype(jnp.float32) + partial.astype(jnp.float32)).astype(x.dtype)
        return output

    return jax.shard_map(
        _run,
        mesh=mesh,
        in_specs=(
            token_spec,
            P(),
            P(),
            P(),
            None if w1_scale is None else P(),
            None if w3_scale is None else P(),
            None if w2_scale is None else P(),
        ),
        out_specs=token_spec,
        check_vma=False,
    )(
        hidden_states,
        w1,
        w3,
        w2,
        w1_scale,
        w3_scale,
        w2_scale,
    )


class FusedEPMoERS(FusedEPMoEV2):
    """Stage-aware GLM MoE: fused-RS prefill and fused-v2 decode.

    Both paths consume the same ``w1``/``w3``/``w2`` parameters. The RS kernel
    receives gate/up separately and joins only the current VMEM tile, avoiding a
    full HBM ``concatenate`` or a duplicated merged checkpoint parameter.
    """

    def __init__(
        self,
        *args,
        fp8_hidden_all_gather: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fp8_hidden_all_gather = fp8_hidden_all_gather

    def _shared_expert_for_rs(
        self,
        hidden_states: jax.Array,
        *,
        enable_act_quant: bool,
        v2_shared_block_size: int = 1024,
    ) -> jax.Array:
        w1_scale = self.w1_shared_scale.value if self.w1_shared_scale is not None else None
        w3_scale = self.w3_shared_scale.value if self.w3_shared_scale is not None else None
        w2_scale = self.w2_shared_scale.value if self.w2_shared_scale is not None else None
        return fused_rs_shared_expert(
            hidden_states,
            self.w1_shared.value,
            self.w3_shared.value,
            self.w2_shared.value,
            w1_scale=w1_scale,
            w3_scale=w3_scale,
            w2_scale=w2_scale,
            activation_quantized_dtype=(
                self.activation_quantized_dtype if enable_act_quant else None
            ),
            mesh=self.mesh,
            v2_shared_block_size=v2_shared_block_size,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        use_rs: bool,
        block_config=None,
        out_sharding: jax.sharding.Sharding | None = None,
        swiglu_limit: float | None = None,
        shared_swiglu_limit: float | None = None,
    ) -> jax.Array:
        if not use_rs:
            return super().__call__(
                hidden_states,
                topk_weights,
                topk_ids,
                block_config=block_config,
                out_sharding=out_sharding,
                swiglu_limit=swiglu_limit,
                shared_swiglu_limit=shared_swiglu_limit,
            )

        if swiglu_limit is not None or shared_swiglu_limit is not None:
            raise ValueError("fused_rs currently supports GLM SiLU without SwiGLU limits")

        from sgl_jax.srt.kernels.fused_moe.fused_rs import fused_moe_func_rs

        w1_scale = self.w1_scale.value if self.w1_scale is not None else None
        w3_scale = self.w3_scale.value if self.w3_scale is not None else None
        w2_scale = self.w2_scale.value if self.w2_scale is not None else None
        expert_axis = ("data", "tensor")
        rs_input_sharding = NamedSharding(self.mesh, P(expert_axis, None))
        rs_hidden_states = jax.sharding.reshard(hidden_states, rs_input_sharding)
        rs_topk_weights = jax.sharding.reshard(topk_weights, rs_input_sharding)
        rs_topk_ids = jax.sharding.reshard(topk_ids, rs_input_sharding)

        routed_output = fused_moe_func_rs(
            hidden_states=rs_hidden_states,
            w1=self.w1.value,
            w3=self.w3.value,
            w2=self.w2.value,
            w1_scale=w1_scale,
            w3_scale=w3_scale,
            w2_scale=w2_scale,
            w1_bias=None,
            w2_bias=None,
            gating_output=None,
            topk=self.num_experts_per_tok,
            renormalize=False,
            mesh=self.mesh,
            activation=self.activation,
            scoring_fn="softmax",
            topk_weights=rs_topk_weights,
            topk_indices=rs_topk_ids,
            fp8_hidden_all_gather=self.fp8_hidden_all_gather,
        )

        if self.w1_shared is not None and not self.disable_shared_expert:
            enable_act_quant = (
                self.activation_quantized_dtype is not None
                and self.enable_act_quant_cfg is not False
                and self.w1_shared_scale is not None
            )
            shared_output = self._shared_expert_for_rs(
                rs_hidden_states,
                enable_act_quant=enable_act_quant,
            )
            routed_output = (
                routed_output.astype(jnp.float32) + shared_output.astype(jnp.float32)
            ).astype(hidden_states.dtype)

        target = out_sharding or jax.sharding.NamedSharding(self.mesh, P("data", None))
        return jax.sharding.reshard(routed_output, target)
